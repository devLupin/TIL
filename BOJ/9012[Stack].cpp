#include <iostream>
#include <stack>
#include <string>
using namespace std;

bool solve(string str)
{
	stack<char> s;
	for (int i = 0; i < str.size(); i++) {
		if (str[i] == '(')
			s.push('(');
		else {
			if (s.empty()) return false;
			else s.pop();
		}
	}
	if (s.empty()) return true;
	return false;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	string str;
	int n; cin >> n;
	while (n--) {
		string str;
		cin >> str;

		cout << (solve(str) ? "YES" : "NO") << '\n';
	}
}