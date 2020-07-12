#include <iostream>
#include <queue>
using namespace std;
typedef pair<int, int> pi;

struct cmp {
	bool operator()(pi a, pi b) {
		return a.second < b.second;
	}
};

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int tc; cin >> tc;
	while (tc--) {
		priority_queue<int> pq;
		queue<pi> q;
		int n, m; cin >> n >> m;

		for (int i = 0; i < n; i++) {
			int a; cin >> a;
			q.push({ i,a });
			pq.push(a);
		}

		int cnt = 0;
		while (!q.empty()) {
			pi cur = q.front();
			q.pop();

			if (cur.second == pq.top()) {
				pq.pop();
				cnt++;

				if (cur.first == m) {
					cout << cnt << '\n';
					break;
				}
			}
			else {
				q.push(cur);
			}
		}
	}
}