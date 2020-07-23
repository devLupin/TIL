#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

typedef pair<int, int> pi;
typedef struct {
	int y, x, dist;
}pos;

pi src;
queue<pi> fire;
int n, m;
char map[1001][1001];
bool visited[1001][1001];

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

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
				map[ny][nx] = '*';
				fire.push({ ny,nx });
			}
		}
	}
}

int bfs()
{
	memset(visited, false, sizeof(visited));

	queue<pos> q;
	q.push({ src.first, src.second, 0 });
	visited[src.first][src.second] = true;

	while (!q.empty())
	{
		spread();
		int q_sz = q.size();

		while (q_sz--) {
			pos cur = q.front();
			q.pop();

			if (cur.y == 0 || cur.x == 0 || cur.y == n - 1 || cur.x == m - 1)
				return cur.dist + 1;

			for (int dir = 0; dir < 4; dir++) {
				int ny = cur.y + dy[dir];
				int nx = cur.x + dx[dir];

				if (ny < 0 || nx < 0 || ny >= n || nx >= m)
					continue;
				if (map[ny][nx] == '.' && !visited[ny][nx]) {
					q.push({ ny,nx,cur.dist + 1 });
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
	
	int tc; cin >> tc;
	while (tc--) {
		while (!fire.empty())
			fire.pop();

		cin >> m >> n;

		for (int y = 0; y < n; y++) {
			for (int x = 0; x < m; x++) {
				cin >> map[y][x];
				if (map[y][x] == '@') {
					src = { y,x };
					map[y][x] = '.';
				}
				else if (map[y][x] == '*')
					fire.push({ y,x });
			}
		}

		int ans = bfs();
		(ans > 0) ? (cout << ans << '\n') : (cout << "IMPOSSIBLE" << '\n');
	}
}