#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

int k, m, n;
int map[200][200];
bool visited[200][200][30];

typedef struct {
	int y, x, w, k;
}pos;

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };
const int horse_dy[] = { 1,2,2,1,-1,-2,-2,-1 };
const int horse_dx[] = { -2,-1,1,2,2,1,-1,-2 };

int bfs()
{
	int ret = 0x3f3f3f;

	queue<pos> q;
	q.push({ 0,0,0,0 });
	visited[0][0][0] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		if (cur.y == n - 1 && cur.x == m - 1)
			ret = (ret < cur.w) ? ret : cur.w;

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= m)
				continue;
			if (!visited[ny][nx][cur.k] && map[ny][nx] == 0) {
				q.push({ ny,nx,cur.w + 1,cur.k });
				visited[ny][nx][cur.k] = true;
			}
		}
		if (cur.k != k) {
			for (int dir = 0; dir < 8; dir++) {
				int ny = cur.y + horse_dy[dir];
				int nx = cur.x + horse_dx[dir];

				if (ny < 0 || nx < 0 || ny >= n || nx >= m)
					continue;
				if (!visited[ny][nx][cur.k + 1] && map[ny][nx] == 0) {
					q.push({ ny,nx,cur.w + 1,cur.k + 1 });
					visited[ny][nx][cur.k + 1] = true;
				}
			}
		}
	}
	return ret;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> k >> m >> n;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++)
			cin >> map[y][x];

	memset(visited, false, sizeof(visited));
	int temp = bfs();
	(temp == 0x3f3f3f) ? (cout << -1) : (cout << temp);
}