#include <iostream>
#include <queue>
using namespace std;

int n, m;
char map[1000][1000] = { 0, };
bool visited[1000][1000] = { false, };

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

typedef pair<int, int> pi;
typedef struct {
	int y, x, dist;
}pos;

queue<pi> fire;
pi src;

void spread()
{
	int q_sz = fire.size();
	while (q_sz--) {
		pi cur = fire.front();
		fire.pop();

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.first + dy[dir];
			int nx = cur.second + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= m)
				continue;
			if (map[ny][nx] == '.') {
				map[ny][nx] = 'F';
				fire.push({ ny,nx });
			}
		}
	}
}

int bfs()
{
	queue<pos> q;
	q.push({ src.first, src.second, 0 });
	visited[src.first][src.second] = true;

	while (!q.empty()) {
		spread();
		int q_sz = q.size();
		while (q_sz--) {
			pos cur = q.front();
			q.pop();

			if (cur.y == n - 1 || cur.x == m - 1 || cur.y == 0 || cur.x == 0)
				return cur.dist + 1;

			for (int dir = 0; dir < 4; dir++) {
				int ny = cur.y + dy[dir];
				int nx = cur.x + dx[dir];

				if (ny < 0 || nx < 0 || ny >= n || nx >= m)
					continue;
				if (!visited[ny][nx] && map[ny][nx] == '.') {
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

	cin >> n >> m;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < m; x++) {
			cin >> map[y][x];

			if (map[y][x] == 'J') {
				src = { y,x };
				map[y][x] = '.';
			}
			else if (map[y][x] == 'F')
				fire.push({ y,x });
		}

	int ans = bfs();
	(ans > 0) ? cout << ans : cout << "IMPOSSIBLE";
}