// make_gaddag.cpp -- build a Quackle .gaddag from an existing .dawg, without Qt.
//
// Quackle's own makegaddag tool needs Qt and a plain-text word list. This one
// reads the word list straight out of the .dawg that Quackle ships, builds the
// GADDAG with the same algorithm as quackleio/gaddagfactory.cpp, and copies the
// .dawg's hash into the header so Quackle accepts the pair as matching.
//
// Usage:  make_gaddag data/lexica/nwl23.dawg data/lexica/nwl23.gaddag
//
// Quackle letter codes: letters are stored as (code & 63) + FIRST_LETTER in the
// DAWG; the GADDAG stores the same codes, with 0 as the "^" separator.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kFirstLetter = 5;                  // QUACKLE_FIRST_LETTER
constexpr unsigned char kSeparator = 5 + 55;     // FIRST_LETTER + MAXIMUM_ALPHABET_SIZE
constexpr unsigned char kNullMark = 0;           // separator as written to disk

// ----------------------------------------------------------------- DAWG reader

struct Dawg
{
	std::vector<unsigned char> nodes;  // 7 bytes per node, node 0 unused, root children at 1
	char hash[16];

	void at(int i, unsigned &p, unsigned char &letter, bool &terminal, bool &lastchild) const
	{
		const unsigned char *n = &nodes[size_t(i) * 7];
		p = (n[0] << 16) | (n[1] << 8) | n[2];
		unsigned char l = n[3];
		lastchild = (l & 64) != 0;
		letter = (l & 63) + kFirstLetter;
		unsigned playability = (n[4] << 16) | (n[5] << 8) | n[6];
		terminal = playability != 0;
	}
};

bool readDawg(const char *path, Dawg &d)
{
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	int version = f.get();
	if (version != 1)
	{
		std::cerr << "expected a version-1 .dawg, got version " << version << "\n";
		return false;
	}
	f.read(d.hash, 16);
	// Same header layout as Quackle's V1LexiconInterpreter::loadDawg:
	// 3 bytes, alphabet size, then that many space-terminated UTF-8 letters.
	char skip[3];
	f.read(skip, 3);
	int alphabetSize = f.get();
	for (int i = 0; i < alphabetSize; ++i)
	{
		std::string letter;
		f >> letter;
		f.get();
	}
	d.nodes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	return !d.nodes.empty();
}

void collect(const Dawg &d, int i, std::string &prefix, std::vector<std::string> &out)
{
	while (true)
	{
		unsigned p; unsigned char c; bool t, last;
		d.at(i, p, c, t, last);
		prefix.push_back(char(c));
		if (t) out.push_back(prefix);
		if (p != 0) collect(d, int(p), prefix, out);
		prefix.pop_back();
		if (last) break;
		++i;
	}
}

// --------------------------------------------- GADDAG builder (as gaddagfactory)

struct Node
{
	unsigned char c = 0;
	bool t = false;
	bool lastchild = false;
	int pointer = 0;
	std::vector<Node> children;

	void push(const std::string &w, size_t k = 0)
	{
		if (k == w.size()) { t = true; return; }
		const unsigned char first = (unsigned char)w[k];
		// Input is sorted, so a matching child, if any, is the last one.
		if (children.empty() || children.back().c != first)
		{
			children.emplace_back();
			children.back().c = first;
		}
		children.back().push(w, k + 1);
	}

	void layout(std::vector<Node *> &list)
	{
		if (!children.empty())
		{
			pointer = int(list.size());
			children.back().lastchild = true;
		}
		for (auto &ch : children) list.push_back(&ch);
		for (auto &ch : children) ch.layout(list);
	}
};

} // namespace

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cerr << "usage: make_gaddag input.dawg output.gaddag\n";
		return 2;
	}

	Dawg dawg;
	if (!readDawg(argv[1], dawg))
	{
		std::cerr << "could not read " << argv[1] << "\n";
		return 1;
	}

	std::vector<std::string> words;
	std::string prefix;
	collect(dawg, 1, prefix, words);
	std::cerr << "words read from dawg: " << words.size() << "\n";

	// Every word "abcd" becomes: a, ba^cd... -> rev(prefix) + ^ + suffix, for each split.
	std::vector<std::string> g;
	g.reserve(words.size() * 9);
	for (const auto &w : words)
		for (size_t i = 1; i <= w.size(); ++i)
		{
			std::string s(w.rbegin() + (w.size() - i), w.rend());
			if (i < w.size())
			{
				s.push_back(char(kSeparator));
				s.append(w, i, std::string::npos);
			}
			g.push_back(std::move(s));
		}
	std::sort(g.begin(), g.end(), [](const std::string &a, const std::string &b) {
		return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
			[](char x, char y) { return (unsigned char)x < (unsigned char)y; });
	});
	g.erase(std::unique(g.begin(), g.end()), g.end());

	Node root;
	root.c = kNullMark;
	root.lastchild = true;
	for (const auto &s : g) root.push(s);

	std::vector<Node *> list{&root};
	root.layout(list);
	std::cerr << "gaddag nodes: " << list.size() << "\n";

	std::ofstream out(argv[2], std::ios::binary);
	if (!out)
	{
		std::cerr << "could not write " << argv[2] << "\n";
		return 1;
	}
	out.put(1);                 // GADDAG format version 1
	out.write(dawg.hash, 16);   // must equal the .dawg's hash
	for (size_t i = 0; i < list.size(); ++i)
	{
		unsigned p = unsigned(list[i]->pointer);
		if (p != 0) p -= unsigned(i);                  // stored as relative offset
		if (p > 0xFFFFFF)
		{
			std::cerr << "node offset overflow: word list too large for this format\n";
			return 1;
		}
		unsigned char n4 = list[i]->c == kSeparator ? kNullMark : list[i]->c;
		if (list[i]->t) n4 |= 64;
		if (list[i]->lastchild) n4 |= 128;
		const char bytes[4] = {char((p >> 16) & 0xFF), char((p >> 8) & 0xFF), char(p & 0xFF), char(n4)};
		out.write(bytes, 4);
	}
	std::cerr << "wrote " << argv[2] << "\n";
	return 0;
}
