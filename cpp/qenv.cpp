// qenv.cpp -- a small Python bridge to the Quackle Scrabble engine.
//
// This is the ONLY C++ file you need to touch. It does three things:
//   1. loads the lexicon and strategy files once (init),
//   2. wraps a Quackle::Game in a Python class (Game),
//   3. converts Quackle's data (board, racks, move lists) into numpy arrays.
//
// Letter encoding used everywhere on the Python side:
//   0 = blank (on a rack) / empty square (on the board)
//   1..26 = A..Z
// Blank-designated letters on the board keep their letter (1..26); a separate
// boolean array says which squares hold a blank.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "alphabetparameters.h"
#include "bag.h"
#include "board.h"
#include "boardparameters.h"
#include "datamanager.h"
#include "game.h"
#include "gameparameters.h"
#include "generator.h"
#include "lexiconparameters.h"
#include "move.h"
#include "rack.h"
#include "strategyparameters.h"
#include "test/trademarkedboards.h"   // the standard 15x15 board layout

namespace py = pybind11;

namespace {

constexpr int kAlpha = 27;      // blank + A..Z
constexpr int kMaxPlaced = 7;   // most tiles one move can place

// Quackle's singleton that owns the lexicon, alphabet, board layout and RNG.
// Created once by init() and kept alive for the life of the process.
Quackle::DataManager *g_dm = nullptr;

void requireInit()
{
	if (!g_dm)
		throw std::runtime_error("call qenv.init(...) before creating a Game");
}

// Quackle letter code -> our 0..26 code (blank flag returned separately).
inline int toIndex(Quackle::Letter letter, bool *isBlank = nullptr)
{
	const Quackle::AlphabetParameters *ap = QUACKLE_ALPHABET_PARAMETERS;
	if (letter == QUACKLE_BLANK_MARK)
	{
		if (isBlank) *isBlank = true;
		return 0;
	}
	bool blank = ap->isBlankLetter(letter);
	if (isBlank) *isBlank = blank;
	if (blank) letter = ap->clearBlankness(letter);
	return static_cast<int>(letter - QUACKLE_FIRST_LETTER) + 1;
}

// Count tiles of a LetterString into a length-27 vector (index 0 = blank).
template <class S>
void countInto(const S &letters, uint8_t *out)
{
	for (auto ch : letters)
	{
		Quackle::Letter l = static_cast<Quackle::Letter>(ch);
		if (l == QUACKLE_PLAYED_THRU_MARK || l == QUACKLE_NULL_MARK) continue;
		bool blank = false;
		int idx = toIndex(l, &blank);
		// A blank used as a letter is still a blank tile on the rack.
		out[blank ? 0 : idx] += 1;
	}
}

std::string visible(const Quackle::LetterString &s)
{
	return QUACKLE_ALPHABET_PARAMETERS->userVisible(s);
}

// Cross-check bitset -> 26-bit mask: bit k set <=> letter A+k may go here.
inline uint32_t crossMask(const LetterBitset &b)
{
	uint32_t m = 0;
	for (int k = 0; k < 26; ++k)
		if (b.test(k)) m |= (1u << k);
	return m;
}

// Write one board (letters, blanks, cross-checks) into slot k of batched arrays.
void writeBoard(const Quackle::Board &b, py::ssize_t k,
                py::detail::unchecked_mutable_reference<uint8_t, 3> &L,
                py::detail::unchecked_mutable_reference<bool, 3> &B,
                py::detail::unchecked_mutable_reference<uint32_t, 3> &CH,
                py::detail::unchecked_mutable_reference<uint32_t, 3> &CV)
{
	for (int r = 0; r < b.height(); ++r)
		for (int c = 0; c < b.width(); ++c)
		{
			Quackle::Letter l = b.letter(r, c);
			const bool filled = l != QUACKLE_NULL_MARK;
			L(k, r, c) = filled ? toIndex(l) : 0;
			B(k, r, c) = filled && b.isBlank(r, c);
			CH(k, r, c) = crossMask(b.hcross(r, c));
			CV(k, r, c) = crossMask(b.vcross(r, c));
		}
}

} // namespace

// ---------------------------------------------------------------------------

void init(const std::string &dataDir, const std::string &lexicon, py::object seed)
{
	if (g_dm)
		throw std::runtime_error("qenv.init() was already called");

	g_dm = new Quackle::DataManager;  // never freed: Quackle expects a global
	g_dm->setAppDataDirectory(dataDir);
	// Standard 15x15 premium-square layout. (Quackle keeps it in test/ for
	// trademark reasons; Quackle::EnglishBoard is a deliberately odd 13x17 board.)
	g_dm->setBoardParameters(new ScrabbleBoard());
	// Fall back to the English leave/strategy tables when the lexicon has none.
	g_dm->setBackupLexicon("default_english");

	const std::string dawg = Quackle::LexiconParameters::findDictionaryFile(lexicon + ".dawg");
	g_dm->lexiconParameters()->loadDawg(dawg);
	// A GADDAG is optional; Quackle generates moves faster when one is present.
	const std::string gaddag = Quackle::LexiconParameters::findDictionaryFile(lexicon + ".gaddag");
	g_dm->lexiconParameters()->loadGaddag(gaddag);
	if (!g_dm->lexiconParameters()->hasDawg() && !g_dm->lexiconParameters()->hasGaddag())
		throw std::runtime_error("could not load lexicon '" + lexicon + "' from " + dataDir + "/lexica");

	g_dm->strategyParameters()->initialize(lexicon);

	if (!seed.is_none())
		g_dm->seedRandomNumbers(seed.cast<unsigned int>());
}

void seed(unsigned int s)
{
	requireInit();
	g_dm->seedRandomNumbers(s);
}

// Premium-square layout: 0 none, 1 double letter, 2 triple letter,
// 3 double word, 4 triple word. (Covered squares lose their premium in play.)
py::array_t<int8_t> premiums()
{
	requireInit();
	const Quackle::BoardParameters *bp = QUACKLE_BOARD_PARAMETERS;
	const int H = bp->height(), W = bp->width();
	py::array_t<int8_t> out({H, W});
	auto P = out.mutable_unchecked<2>();
	for (int r = 0; r < H; ++r)
		for (int c = 0; c < W; ++c)
		{
			const int lm = bp->letterMultiplier(r, c), wm = bp->wordMultiplier(r, c);
			P(r, c) = wm == 3 ? 4 : wm == 2 ? 3 : lm == 3 ? 2 : lm == 2 ? 1 : 0;
		}
	return out;
}

bool hasGaddag()
{
	requireInit();
	return g_dm->lexiconParameters()->hasGaddag();
}

// ---------------------------------------------------------------------------

class Game
{
public:
	Game() { requireInit(); reset(); }

	void reset()
	{
		m_game = std::make_unique<Quackle::Game>();
		Quackle::PlayerList players;
		players.push_back(Quackle::Player("P0", Quackle::Player::HumanPlayerType, 0));
		players.push_back(Quackle::Player("P1", Quackle::Player::HumanPlayerType, 1));
		m_game->setPlayers(players);
		m_game->addPosition();  // shuffles the bag and deals both racks
		m_moves.clear();
	}

	Game copy() const
	{
		Game g(NoInit{});
		g.m_game = std::make_unique<Quackle::Game>(*m_game);
		g.m_moves = m_moves;
		return g;
	}

	// ----- state -----------------------------------------------------------

	int toMove() const { return pos().currentPlayer().id(); }
	bool gameOver() const { return pos().gameOver(); }
	int turn() const { return pos().turnNumber(); }
	int bagSize() const { return pos().bag().size(); }
	int scorelessTurns() const { return pos().scorelessTurnsInARow(); }

	py::tuple scores() const
	{
		int s[2] = {0, 0};
		for (const auto &p : pos().players())
			if (p.id() == 0 || p.id() == 1) s[p.id()] = p.score();
		return py::make_tuple(s[0], s[1]);
	}

	// (letters[15,15] uint8 in 0..26, blanks[15,15] bool)
	py::tuple board() const
	{
		const Quackle::Board &b = pos().board();
		const int H = b.height(), W = b.width();
		py::array_t<uint8_t> letters({H, W});
		py::array_t<bool> blanks({H, W});
		auto L = letters.mutable_unchecked<2>();
		auto B = blanks.mutable_unchecked<2>();
		for (int r = 0; r < H; ++r)
			for (int c = 0; c < W; ++c)
			{
				Quackle::Letter l = b.letter(r, c);
				L(r, c) = (l == QUACKLE_NULL_MARK) ? 0 : toIndex(l);
				B(r, c) = (l != QUACKLE_NULL_MARK) && b.isBlank(r, c);
			}
		return py::make_tuple(letters, blanks);
	}

	// Rack as 27 counts. player=None means the player to move.
	py::array_t<uint8_t> rack(py::object player) const
	{
		py::array_t<uint8_t> out(kAlpha);
		std::fill_n(out.mutable_data(), kAlpha, 0);
		countInto(playerById(player).rack().tiles(), out.mutable_data());
		return out;
	}

	std::string rackString(py::object player) const
	{
		return visible(playerById(player).rack().tiles());
	}

	// Tiles the player to move cannot see: the bag plus the opponent's rack.
	py::array_t<uint8_t> unseen() const
	{
		py::array_t<uint8_t> out(kAlpha);
		std::fill_n(out.mutable_data(), kAlpha, 0);
		countInto(pos().unseenBag().tiles(), out.mutable_data());
		return out;
	}

	std::string boardString() const { return pos().board().toString(); }

	// Cross-checks of the current board as two (15,15) uint32 masks.
	// cross_h: letters allowed by the word formed ACROSS through this square
	//          (constrains vertical plays); cross_v: by the word formed DOWN
	//          (constrains horizontal plays). Empty squares with no neighbours
	//          in that direction allow all 26 letters; filled squares are 0.
	py::tuple crossChecks() const
	{
		auto letters = py::array_t<uint8_t>({1, 15, 15});
		auto blanks = py::array_t<bool>({1, 15, 15});
		auto ch = py::array_t<uint32_t>({1, 15, 15});
		auto cv = py::array_t<uint32_t>({1, 15, 15});
		auto L = letters.mutable_unchecked<3>(); auto B = blanks.mutable_unchecked<3>();
		auto CH = ch.mutable_unchecked<3>(); auto CV = cv.mutable_unchecked<3>();
		writeBoard(pos().board(), 0, L, B, CH, CV);
		return py::make_tuple(ch[py::int_(0)], cv[py::int_(0)]);
	}

	// Afterstates for stage 2: for each index into the last legal_moves(), the
	// board after placing that move, BEFORE new tiles are drawn. Exchanges and
	// pass leave the board unchanged. Cross-checks are recomputed for each.
	py::dict afterstates(const std::vector<int> &indices) const
	{
		const py::ssize_t K = static_cast<py::ssize_t>(indices.size());
		const Quackle::Board &now = pos().board();
		const int H = now.height(), W = now.width();
		py::array_t<uint8_t> letters({K, (py::ssize_t)H, (py::ssize_t)W});
		py::array_t<bool> blanks({K, (py::ssize_t)H, (py::ssize_t)W});
		py::array_t<bool> placed({K, (py::ssize_t)H, (py::ssize_t)W});
		py::array_t<uint32_t> ch({K, (py::ssize_t)H, (py::ssize_t)W});
		py::array_t<uint32_t> cv({K, (py::ssize_t)H, (py::ssize_t)W});
		auto L = letters.mutable_unchecked<3>(); auto B = blanks.mutable_unchecked<3>();
		auto N = placed.mutable_unchecked<3>();
		auto CH = ch.mutable_unchecked<3>(); auto CV = cv.mutable_unchecked<3>();

		// Copy the position once without its (possibly huge) move list, so each
		// per-candidate Generator copies only the board, racks and bag.
		Quackle::GamePosition light(pos());
		light.setMoves(Quackle::MoveList());

		for (py::ssize_t k = 0; k < K; ++k)
		{
			const int i = indices[k];
			if (i < 0 || i >= (int)m_moves.size())
				throw std::out_of_range("afterstate index out of range; call legal_moves() first");
			const Quackle::Move &m = m_moves[i];
			if (m.action == Quackle::Move::Place)
			{
				Quackle::Generator gen(light);
				gen.makeMove(m, true);   // places tiles, updates affected cross-checks
				writeBoard(gen.position().board(), k, L, B, CH, CV);
			}
			else
				writeBoard(now, k, L, B, CH, CV);
			for (int r = 0; r < H; ++r)
				for (int c = 0; c < W; ++c)
					N(k, r, c) = (L(k, r, c) != 0) && now.letter(r, c) == QUACKLE_NULL_MARK;
		}

		py::dict d;
		d["letters"] = letters; d["blanks"] = blanks; d["placed"] = placed;
		d["cross_h"] = ch; d["cross_v"] = cv;
		return d;
	}

	// ----- moves -----------------------------------------------------------

	// Generate every legal move for the player to move, plus pass.
	// Returns a dict of numpy arrays, one row per move (see README).
	py::dict legalMoves(int limit)
	{
		Quackle::GamePosition &p = m_game->currentPosition();
		p.kibitz(limit);                      // all plays + exchanges, sorted by equity
		m_moves = p.moves();
		Quackle::Move pass = Quackle::Move::createPassMove();
		p.scoreMove(pass);
		pass.equity = p.calculateEquity(pass);
		m_moves.push_back(pass);

		const py::ssize_t n = static_cast<py::ssize_t>(m_moves.size());
		py::array_t<int8_t> kind(n), row(n), col(n), ntiles(n);
		py::array_t<bool> horiz(n), bingo(n);
		py::array_t<int16_t> score(n);
		py::array_t<float> equity(n);
		py::array_t<int8_t> tiles({n, (py::ssize_t)kMaxPlaced, (py::ssize_t)4});
		py::array_t<uint8_t> leave({n, (py::ssize_t)kAlpha});

		auto K = kind.mutable_unchecked<1>(); auto R = row.mutable_unchecked<1>();
		auto C = col.mutable_unchecked<1>(); auto N = ntiles.mutable_unchecked<1>();
		auto Hz = horiz.mutable_unchecked<1>(); auto Bg = bingo.mutable_unchecked<1>();
		auto S = score.mutable_unchecked<1>(); auto E = equity.mutable_unchecked<1>();
		auto T = tiles.mutable_unchecked<3>(); auto Lv = leave.mutable_unchecked<2>();

		const Quackle::LetterString rackTiles = p.currentPlayer().rack().tiles();

		for (py::ssize_t i = 0; i < n; ++i)
		{
			const Quackle::Move &m = m_moves[i];
			for (int j = 0; j < kMaxPlaced; ++j)
				for (int k = 0; k < 4; ++k) T(i, j, k) = -1;

			if (m.action == Quackle::Move::Place)        K(i) = 0;
			else if (m.action == Quackle::Move::Exchange) K(i) = 1;
			else                                          K(i) = 2;

			R(i) = (K(i) == 0) ? m.startrow : -1;
			C(i) = (K(i) == 0) ? m.startcol : -1;
			Hz(i) = m.horizontal;
			Bg(i) = m.isBingo;
			S(i) = static_cast<int16_t>(m.score);
			E(i) = static_cast<float>(m.equity);

			// Newly placed tiles only (skip squares the word plays through).
			int placed = 0;
			if (K(i) == 0)
			{
				const Quackle::LetterString &t = m.tiles();
				for (size_t j = 0; j < t.length(); ++j)
				{
					Quackle::Letter l = t[j];
					if (l == QUACKLE_PLAYED_THRU_MARK) continue;
					bool blank = false;
					int idx = toIndex(l, &blank);
					int rr = m.horizontal ? m.startrow : m.startrow + (int)j;
					int cc = m.horizontal ? m.startcol + (int)j : m.startcol;
					if (placed < kMaxPlaced)
					{
						T(i, placed, 0) = rr; T(i, placed, 1) = cc;
						T(i, placed, 2) = idx; T(i, placed, 3) = blank;
					}
					++placed;
				}
			}
			else if (K(i) == 1)
				placed = static_cast<int>(m.tiles().length());
			N(i) = placed;

			// Leave = rack minus the tiles this move uses.
			uint8_t rc[kAlpha] = {0};
			countInto(rackTiles, rc);
			uint8_t used[kAlpha] = {0};
			if (K(i) != 2) countInto(m.usedTiles(), used);
			for (int a = 0; a < kAlpha; ++a)
				Lv(i, a) = (rc[a] >= used[a]) ? rc[a] - used[a] : 0;
		}

		py::dict d;
		d["kind"] = kind; d["row"] = row; d["col"] = col; d["horizontal"] = horiz;
		d["score"] = score; d["equity"] = equity; d["ntiles"] = ntiles;
		d["bingo"] = bingo; d["tiles"] = tiles; d["leave"] = leave;
		return d;
	}

	std::vector<std::string> moveStrings() const
	{
		std::vector<std::string> out;
		out.reserve(m_moves.size());
		for (const auto &m : m_moves) out.push_back(m.toString());
		return out;
	}

	// Commit move i from the most recent legal_moves() call.
	void play(int i)
	{
		if (gameOver()) throw std::runtime_error("game is over");
		if (i < 0 || i >= (int)m_moves.size())
			throw std::out_of_range("move index out of range; call legal_moves() first");
		Quackle::Move m = m_moves[i];
		m_moves.clear();
		m_game->commitMove(m);
	}

	// Quackle's "Speedy Player": the single best move by static equity
	// (score + leave value + small positional terms). Your baseline.
	std::string playSpeedy()
	{
		if (gameOver()) throw std::runtime_error("game is over");
		Quackle::Move m = m_game->currentPosition().staticBestMove();
		m_moves.clear();
		m_game->commitMove(m);
		return m.toString();
	}

private:
	struct NoInit {};
	explicit Game(NoInit) {}

	const Quackle::GamePosition &pos() const { return m_game->currentPosition(); }

	const Quackle::Player &playerById(py::object player) const
	{
		const Quackle::GamePosition &p = pos();
		if (player.is_none()) return p.currentPlayer();
		const int id = player.cast<int>();
		for (const auto &pl : p.players())
			if (pl.id() == id) return pl;
		throw std::out_of_range("player must be 0, 1 or None");
	}

	std::unique_ptr<Quackle::Game> m_game;
	Quackle::MoveList m_moves;
};

// ---------------------------------------------------------------------------

PYBIND11_MODULE(qenv, m)
{
	m.doc() = "Minimal Python bridge to the Quackle Scrabble engine";

	m.def("init", &init, py::arg("data_dir"), py::arg("lexicon") = "nwl23",
	      py::arg("seed") = py::none(),
	      "Load the lexicon and strategy files. Call once per process.");
	m.def("seed", &seed, py::arg("seed"), "Reseed Quackle's tile-drawing RNG.");
	m.def("premiums", &premiums, "(15,15) int8: 0 none, 1 DL, 2 TL, 3 DW, 4 TW");
	m.def("has_gaddag", &hasGaddag, "True if a GADDAG was loaded (faster move generation).");

	py::class_<Game>(m, "Game")
		.def(py::init<>())
		.def("reset", &Game::reset, "Start a new game: fresh bag, new racks.")
		.def("copy", &Game::copy, "Independent deep copy of the game.")
		.def_property_readonly("to_move", &Game::toMove)
		.def_property_readonly("game_over", &Game::gameOver)
		.def_property_readonly("turn", &Game::turn, "Round number: both players' moves in a round share it (Quackle convention).")
		.def_property_readonly("bag_size", &Game::bagSize)
		.def_property_readonly("scoreless_turns", &Game::scorelessTurns)
		.def("scores", &Game::scores)
		.def("board", &Game::board, "(letters[15,15] uint8, blanks[15,15] bool)")
		.def("rack", &Game::rack, py::arg("player") = py::none())
		.def("rack_string", &Game::rackString, py::arg("player") = py::none())
		.def("unseen", &Game::unseen, "27 counts: bag + opponent rack")
		.def("board_string", &Game::boardString)
		.def("cross_checks", &Game::crossChecks, "(cross_h, cross_v): (15,15) uint32 letter masks, bit k = letter A+k")
		.def("afterstates", &Game::afterstates, py::arg("indices"),
		     "Boards after the given moves (from the last legal_moves()), before drawing.")
		.def("legal_moves", &Game::legalMoves, py::arg("limit") = 100000,
		     "All legal moves (incl. exchanges and pass) as a dict of numpy arrays.")
		.def("move_strings", &Game::moveStrings, "Human-readable form of the last legal_moves().")
		.def("play", &Game::play, py::arg("index"))
		.def("play_speedy", &Game::playSpeedy);
}
