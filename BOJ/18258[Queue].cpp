#include <iostream>
#include <queue>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	deque<int> dq;

	int tc; cin >> tc;
	while (tc--) {
		string opt; cin >> opt;
		int temp;
		if (opt == "push") {
			cin >> temp;
			dq.push_back(temp);
		}
		else if (opt == "pop") {
			if (!dq.empty()) { 
				cout << dq.front() << '\n'; 
				dq.pop_front(); 
			}
			else cout << -1 << '\n';
		}
		else if (opt == "size")
			cout << dq.size() << '\n';
		else if (opt == "empty") {
			if (dq.empty()) cout << 1 << '\n';
			else cout << 0 << '\n';
		}
		else if (opt == "front") {
			if (!dq.empty()) cout << dq.front() << '\n';
			else cout << -1 << '\n';
		}
		else if (opt == "back") {
			if (!dq.empty()) cout << dq.back() << '\n';
			else cout << -1 << '\n';
		}
	}
}
