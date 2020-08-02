#include <iostream>
#include <queue>
using namespace std;

int map[1000][1000], n, m, src_x, src_y, dest_x, dest_y;
bool visited[1000][1000][2] = { false, };
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

typedef struct {
	int y, x, cost;
	int wall;
}pos;

int bfs()
{
	queue<pos> q;
	q.push({ src_y, src_x, 0, 1 });
	visited[src_y][src_x][0] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		if (cur.y == dest_y && cur.x == dest_x)
			return cur.cost;

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= m)
				continue;
			if (cur.wall == 1 && !visited[ny][nx][cur.wall - 1] && map[ny][nx] == 1) {
				visited[ny][nx][cur.wall - 1] = true;
				q.push({ ny,nx,cur.cost + 1, cur.wall - 1 });
			}
			else if (!visited[ny][nx][cur.wall] && map[ny][nx] == 0) {
				visited[ny][nx][cur.wall] = true;
				q.push({ ny,nx,cur.cost + 1, cur.wall });
			}
		}
	}
	return -1;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> m;
	cin >> src_y >> src_x >> dest_y >> dest_x;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			cin >> map[i][j];

	src_y--, src_x--, dest_y--, dest_x--;

	cout << bfs();
}