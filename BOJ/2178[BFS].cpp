#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

typedef struct {
	int y, x, w;
}pos;

int n, m;
char map[101][101];
bool visited[101][101];

const int dy[] = { -1,1,0,0 };
const int dx[] = { 0,0,-1,1 };

int bfs() 
{
	memset(visited, 0, sizeof(visited));

	queue<pos> q;
	q.push({ 0,0,1 });
	visited[0][0] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		if (cur.y == n - 1 && cur.x == m - 1)
			return cur.w;

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= m)
				continue;

			if (!visited[ny][nx] && map[ny][nx] == '1') {
				q.push({ ny,nx, cur.w + 1 });
				visited[ny][nx] = true;
			}
		}
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> m;

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			cin >> map[i][j];

	cout << bfs();
}