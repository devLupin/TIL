#include <iostream>
using namespace std;

int r, c, ans = 0;
char map[20][20];
bool visited[26] = { false, };
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

int max(int a, int b) { return (a > b) ? a : b; }

void dfs(int cur_y, int cur_x, int cnt)
{
	if (visited[map[cur_y][cur_x] - 'A'])
		return;
	visited[map[cur_y][cur_x] - 'A'] = true;

	for (int dir = 0; dir < 4; dir++) {
		int ny = cur_y + dy[dir];
		int nx = cur_x + dx[dir];

		if (ny < 0 || nx < 0 || ny >= r || nx >= c)
			continue;
		if (visited[map[ny][nx] - 'A'])
			continue;

		dfs(ny, nx, cnt + 1);
	}
	visited[map[cur_y][cur_x] - 'A'] = false;
	ans = max(ans, cnt);
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> r >> c;
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			cin >> map[i][j];

	dfs(0, 0, 1);
	cout << ans;
}