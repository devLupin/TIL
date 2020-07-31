#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };
int n, m, map[100][100] = { 0, }, last_cheese = 0, time = 0;
bool visited[100][100] = { false, };
typedef pair<int, int> pii;

int check()
{
	int ret = 0;

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			if (map[i][j] != 0)
				ret++;

	return ret;
}

void bfs()
{
	memset(visited, false, sizeof(visited));

	queue<pii> q;
	q.push({ 0,0 });
	visited[0][0] = true;

	while (!q.empty())
	{
		pii cur = q.front();
		q.pop();

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.first + dy[dir];
			int nx = cur.second + dx[dir];

			if (ny < 0 || nx < 0 || ny > n || nx > m)
				continue;
			if (visited[ny][nx])
				continue;
			if (map[ny][nx] > 0) {
				visited[ny][nx] = true;
				map[ny][nx] = map[ny][nx] - 1;
			}
			else {
				q.push({ ny,nx });
				visited[ny][nx] = true;
			}
		}
	}
}

void solve()
{
	while (true) {
		int numOfCheese = check();

		if (numOfCheese <= 0) break;

		last_cheese = numOfCheese;
		bfs();
		time++;
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	cin >> n >> m;

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			cin >> map[i][j];

	solve();
	cout << time << '\n' << last_cheese;
}