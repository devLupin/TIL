#include <queue>
#include <iostream>
#include <vector>
#include <cstring>
using namespace std;

typedef struct {
	int y, x;
}pos;

int n, m;
int map[8][8];
bool visited[8][8];

int ans = -1;

const int dy[] = { -1,1,0,0 };
const int dx[] = { 0,0,-1,1 };

int bfs()
{
	int ret = 0;
	queue<pos> q;

	int copy[8][8];
	bool chk[8][8];
	memset(chk, false, sizeof(chk));

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++) {
			copy[i][j] = map[i][j];

			if (copy[i][j] == 2) {
				q.push({ i,j });
				chk[i][j] = true;
			}
		}

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= m)
				continue;
			if (!chk[ny][nx] && copy[ny][nx] == 0) {
				copy[ny][nx] = 2;
				chk[ny][nx] = true;
				q.push({ ny,nx });
			}
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (copy[i][j] == 0)
				ret++;
		}
	}

	return ret;
}

void dfs(int cnt)
{
	if (cnt == 3) {
		int cmp = bfs();
		ans = (ans > cmp) ? ans : cmp;
		return;
	}

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			if (!visited[i][j] && map[i][j] == 0) {
				visited[i][j] = true;
				map[i][j] = 1;
				dfs(cnt + 1);
				visited[i][j] = false;
				map[i][j] = 0;
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

	memset(visited, false, sizeof(visited));
	dfs(0);

	cout << ans;
}