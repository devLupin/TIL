#include <iostream>
#include <queue>
using namespace std;

queue<int> q;
int n, a, b, ans = 0;
int solve()
{
	while (true) {
		int i, j;
		int q_sz = q.size();
		while (q_sz > 0) {
			if (q_sz >= 2) {
				i = q.front(); q.pop();
				j = q.front(); q.pop();
				q_sz -= 2;
			}
			else {
				q.push(q.front());
				q.pop();
				q_sz -= 1;
				continue;
			}

			if (i == a && j == b) {
				return ans + 1;
			}
			if (j == a && i == b) {
				return ans + 1;
			}

			if (i == a || j == a)
				q.push(a);
			else if (i == b || j == b)
				q.push(b);
			else
				q.push(-1);
		}

		ans++;
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> a >> b;
	for (int i = 1; i <= n; i++)
		q.push(i);
	cout << solve();
}