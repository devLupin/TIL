#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

typedef pair<int, int> pi;
int f, s, g, u, d;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> f >> s >> g >> u >> d;

	bool visited[1000001] = { false, };
	queue<pi> q;
	q.push({ s,0 });
	visited[s] = true;

	while (!q.empty())
	{
		pi cur = q.front();
		q.pop();

		if (cur.first == g) {
			cout << cur.second;
			return 0;
		}

		if (!visited[cur.first + u] && cur.first + u <= f) {
			q.push({ cur.first + u, cur.second + 1 });
			visited[cur.first + u] = true;
		}
		if (!visited[cur.first - d] && cur.first - d >= 1) {
			q.push({ cur.first - d, cur.second + 1 });
			visited[cur.first - d] = true;
		}

	}
	cout << "use the stairs";
}