#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

typedef struct {
	int z, y, x;
}pos;

const int dz[] = { -1,1,0,0,0,0 };
const int dy[] = { 0,0,-1,1,0,0 };
const int dx[] = { 0,0,0,0,-1,1 };

int m, n, h;

int map[101][101][101];
bool visited[101][101][101];
queue<pos> q;
int ans = 0;

void bfs()
{
	while (!q.empty()) {
		int q_size = q.size();

		while (q_size--) {
			pos cur = q.front();
			q.pop();

			for (int dir = 0; dir < 6; dir++) {
				int nz = cur.z + dz[dir];
				int ny = cur.y + dy[dir];
				int nx = cur.x + dx[dir];

				if (nz < 0 || ny < 0 || nx < 0 || nz >= h || ny >= n || nx >= m)
					continue;
				if (visited[nz][ny][nx])
					continue;

				if (map[nz][ny][nx] == 0) {
					q.push({ nz,ny,nx });
					visited[nz][ny][nx] = true;

					map[nz][ny][nx] = 1;
				}
			}
		}
		ans++;
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> m >> n >> h;

	for (int z = 0; z < h; z++)
		for (int y = 0; y < n; y++)
			for (int x = 0; x < m; x++) {
				cin >> map[z][y][x];

				if (map[z][y][x] == 1) {
					q.push({ z,y,x });
					visited[z][y][x] = true;
				}
			}

	memset(visited, false, sizeof(visited));
	bfs();

	for(int z = 0; z<h; z++)
		for(int y=0; y <n; y++)
			for(int x=0; x<m; x++)
				if (map[z][y][x] == 0) {
					cout << "-1";
					return 0;
				}

	cout << ans - 1;
}