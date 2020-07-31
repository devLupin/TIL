#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

char map[50][50];
int r, c;
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

typedef struct {
	int y, x, dist;
}pos;

pos src, dest;
queue<pos> water;

void water_bfs()
{
	int sz = water.size();
	while (sz--) {
		pos cur = water.front();
		water.pop();

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= r || nx >= c)
				continue;
			if (map[ny][nx] == '.') {
				map[ny][nx] = '*';
				water.push({ ny,nx,0 });
			}
		}
	}
}

int bfs()
{
	bool visited[50][50] = { false, };
	queue<pos> q;
	q.push(src);
	visited[src.y][src.x] = true;
	
	while (!q.empty()) {
		water_bfs();
		
		int sz = q.size();
		while (sz--) {
			pos cur = q.front();
			q.pop();

			if (map[cur.y][cur.x] == 'D')
				return cur.dist;
			
			for (int dir = 0; dir < 4; dir++) {
				int ny = dy[dir] + cur.y;
				int nx = dx[dir] + cur.x;

				if (ny < 0 || nx < 0 || ny >= r || nx >= c)
					continue;
				if (visited[ny][nx])
					continue;
				if (map[ny][nx] == '.' || map[ny][nx] == 'D') {
					q.push({ ny,nx, cur.dist + 1 });
					visited[ny][nx] = true;
				}
			}
		}
	}
	return -1;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> r >> c;
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++) {
			cin >> map[i][j];
			
			if (map[i][j] == 'S') {
				src = { i,j,0 };
				map[i][j] = '.';
			}
			if (map[i][j] == 'D')
				dest = { i,j,0 };
			if (map[i][j] == '*')
				water.push({ i,j,0 });
		}

	int ans = bfs();
	(ans > 0) ?  cout << ans : cout << "KAKTUS";
}