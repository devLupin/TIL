#include <iostream>
#include <list>
#include <stack>
#include <string>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int tc; cin >> tc;
	while (tc--) {
		stack<char> result;
		stack<char> temp;
		string str, ans; cin >> str;
		
		for (int i = 0; i < str.size(); i++) {
			char cur = str[i];

			if (cur == '<') {
				if (!result.empty()) {
					temp.push(result.top());
					result.pop();
				}
			}
			else if (cur == '>') {
				if (!temp.empty()) {
					result.push(temp.top());
					temp.pop();
				}
			}
			else if (cur == '-') {
				if (!result.empty())
					result.pop();
			}
			else {
				result.push(cur);
			}
		}

		while (!temp.empty()) {
			result.push(temp.top());
			temp.pop();
		}

		while (!result.empty()) {
			ans += result.top();
			result.pop();
		}
		reverse(ans.begin(), ans.end());
		cout << ans << '\n';
	}
}