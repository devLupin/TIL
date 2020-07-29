#include <iostream>
#include <deque>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	deque<int> dq;
	int n; cin >> n;
	while (n--) {
		string opt; cin >> opt;
		int a;
		if (opt == "push_front") {
			cin >> a;
			dq.push_front(a);
		}
		else if (opt == "push_back") {
			cin >> a;
			dq.push_back(a);
		}
		else if (opt == "pop_front") {
			if (dq.empty()) {
				cout << -1;
				continue;
			}

			cout << dq.front() << '\n';
			dq.pop_front();
		}
		else if (opt == "pop_back") {
			if (dq.empty()) {
				cout << -1;
				continue;
			}

			cout << dq.back() << '\n';
			dq.pop_back();
		}
		else if (opt == "size") {
			cout << dq.size() << '\n';
		}
		else if (opt == "empty") {
			cout << ((dq.empty()) ? 1 : 0) << '\n';
		}
		else if (opt == "front") {
			if (dq.empty()) {
				cout << -1;
				continue;
			}

			cout << dq.front() << '\n';
		}
		else if (opt == "back") {
			if (dq.empty()) {
				cout << -1;
				continue;
			}

			cout << dq.back() << '\n';
		}
	}
}