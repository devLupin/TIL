#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

typedef struct {
	int y, x;
}pos;

int n, m;
int map[1000][1000];
bool visited[1000][1000];

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };
queue<pos> q;

int bfs()
{
	int ret = 0;

	while (!q.empty()) {
		int q_sz = q.size();

		while (q_sz--) {
			pos cur = q.front();
			q.pop();

			for (int dir = 0; dir < 4; dir++) {
				int ny = cur.y + dy[dir];
				int nx = cur.x + dx[dir];

				if (ny < 0 || nx < 0 || ny >= n || nx >= m)
					continue;
				if (!visited[ny][nx] && map[ny][nx] == 0) {
					map[ny][nx] = 1;
					visited[ny][nx] = true;
					q.push({ ny,nx });
				}
			}
		}
		ret++;
	}
	return ret;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	memset(visited, false, sizeof(visited));

	cin >> m >> n;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++) {
			cin >> map[y][x];
			
			if (map[y][x] == 1) {
				q.push({ y,x });
				visited[y][x] = true;
			}
		}

	int ans = bfs();

	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++)
			if (map[y][x] == 0) {
				cout << "-1";
				return 0;
			}

	cout << ans - 1;
}