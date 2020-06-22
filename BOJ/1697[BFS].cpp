#include <iostream>
#include <queue>
#include <cstring>
using namespace std;
typedef pair<int, int> pi;

int n, k;
bool visited[100001];
int bfs()
{
	memset(visited, false, sizeof(visited));
	queue<pi> q;
	q.push({ n, 0 });
	visited[n] = true;

	while (!q.empty())
	{
		pi cur = q.front();
		q.pop();

		if (cur.first == k)
			return cur.second;

		if (!visited[cur.first - 1] && cur.first - 1 >= 0 && cur.first - 1 <= 100000) {
			q.push({ cur.first - 1, cur.second + 1 });
			visited[cur.first - 1] = true;
		}
		if (!visited[cur.first + 1] && cur.first + 1 >= 0 && cur.first + 1 <= 100000) {
			q.push({ cur.first + 1, cur.second + 1 });
			visited[cur.first + 1] = true;
		}
		if (!visited[cur.first * 2] && cur.first * 2 >= 0 && cur.first * 2 <= 100000) {
			q.push({ cur.first * 2, cur.second + 1 });
			visited[cur.first * 2] = true;
		}
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> k;
	cout << bfs();
}