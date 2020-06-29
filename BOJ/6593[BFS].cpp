#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

int l, r, c;
char map[30][30][30];
bool visited[30][30][30];

const int dz[] = { -1,1,0,0,0,0 }, dy[] = { 0,0,-1,1,0,0 }, dx[] = { 0,0,0,0,-1,1 };

typedef struct {
	int z, y, x, w;
}pos;

pos src, dest;

int bfs()
{
	int ret = 0;

	queue<pos> q;
	q.push(src);
	visited[src.z][src.y][src.x] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		for (int dir = 0; dir < 6; dir++) {
			int nz = cur.z + dz[dir];
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (nz < 0 || ny < 0 || nx < 0 || nz >= l || ny >= r || nx >= c)
				continue;
			if (nz == dest.z && ny == dest.y && nx == dest.x)
				return cur.w + 1;
			if (!visited[nz][ny][nx] && map[nz][ny][nx] == '.') {
				q.push({ nz,ny,nx,cur.w + 1 });
				visited[nz][ny][nx] = true;
			}
		}
	}
	return -1;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	while (true) {
		cin >> l >> r >> c;
		if (l == 0 && r == 0 && c == 0)
			break;

		for (int z = 0; z < l; z++)
			for (int y = 0; y < r; y++)
				for (int x = 0; x < c; x++) {
					cin >> map[z][y][x];

					if (map[z][y][x] == 'S')
						src = { z,y,x,0 };
					if (map[z][y][x] == 'E')
						dest = { z,y,x,0 };
				}

		memset(visited, false, sizeof(visited));
		int temp = bfs();
		(temp == -1) ? (cout << "Trapped!" << '\n') : (cout << "Escaped in " << temp << " minute(s)." << '\n');
	}
}