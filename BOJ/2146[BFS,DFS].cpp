#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

int map[101][101];
bool visited[101][101];
bool chk[101][101];
int n;

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

typedef struct {
	int y, x, w;
}pos;

void dfs(int y, int x, int part)
{
	visited[y][x] = true;
	map[y][x] = part;

	for (int dir = 0; dir < 4; dir++) {
		int ny = y + dy[dir];
		int nx = x + dx[dir];

		if (ny < 0 || nx < 0 || ny >= n || nx >= n)
			continue;
		if (!visited[ny][nx] && map[ny][nx] == 1)
			dfs(ny, nx, part);
	}
}

int bfs(int y, int x, int part)
{
	memset(chk, false, sizeof(chk));
	queue<pos> q;
	q.push({ y,x,0 });
	chk[y][x] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= n)
				continue;

			if (map[ny][nx] != 0 && map[ny][nx] != part)
				return cur.w;

			if (!chk[ny][nx] && map[ny][nx] == 0) {
				q.push({ ny,nx, cur.w + 1 });
				chk[ny][nx] = true;
			}

		}
	}
	return 0x3f3f3f;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++)
			cin >> map[y][x];

	memset(visited, false, sizeof(visited));
	int part = 1;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++)
			if (!visited[y][x] && map[y][x] == 1) {
				dfs(y, x, part);
				part++;
			}


	memset(visited, false, sizeof(visited));
	int ans = 0x3f3f3f;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++)
			if (!visited[y][x] && map[y][x] != 0) {
				visited[y][x] = true;
				int comp = bfs(y, x, map[y][x]);

				ans = (comp != 0 && comp < ans) ? comp : ans;
			}
	cout << ans;
}