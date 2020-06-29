#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

char map[50][50];
int n, m;

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

typedef struct {
	int y, x, w;
}pos;

int bfs(int sy, int sx)
{
	int ret = -1;
	bool visited[50][50];
	memset(visited, false, sizeof(visited));

	queue<pos> q;
	q.push({ sy,sx, 0 });
	visited[sy][sx] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		ret = (ret < cur.w) ? cur.w : ret;

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= m)
				continue;
			if (!visited[ny][nx] && map[ny][nx] == 'L') {
				q.push({ ny,nx, cur.w + 1 });
				visited[ny][nx] = true;
			}
		}
	}

	return ret;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int ans = -1;
	bool chk[50][50];
	memset(chk, false, sizeof(chk));

	cin >> n >> m;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++)
			cin >> map[y][x];

	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++) {
			if (!chk[y][x] && map[y][x] == 'L') {
				chk[y][x] = true;
				int comp = bfs(y, x);
				ans = (ans < comp) ? comp : ans;
			}
		}

	cout << ans;
}