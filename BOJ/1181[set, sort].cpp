#include <iostream>
#include <set>
#include <string>
using namespace std;

struct comp {
	bool operator() (const string& a, const string& b) const{
		if (a.size() == b.size())
			return a < b;
		return a.size() < b.size();
	}
};

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n; cin >> n;
	set<string,comp> s;

	for (int i = 0; i < n; i++) {
		string str; cin >> str;
		s.insert(str);
	}
	for (auto it = s.begin(); it != s.end(); it++) {
		cout << (*it) << '\n';
	}
}