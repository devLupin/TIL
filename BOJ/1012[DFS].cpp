#include <iostream>
#include <queue>
#include <vector>
#include <cstring>
using namespace std;

int T, n, m, k;
int map[51][51];
bool visited[51][51];
vector<pair<int, int>> v;

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

void dfs(int y, int x)
{
	visited[y][x] = true;

	for (int dir = 0; dir < 4; dir++) {
		int ny = y + dy[dir];
		int nx = x + dx[dir];

		if (ny < 0 || nx < 0 || ny >= n || nx >= m)
			continue;
		if (!visited[ny][nx] && map[ny][nx] == 1) {
			visited[ny][nx] = true;
			dfs(ny, nx);
		}
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> T;
	while (T--) {
		memset(visited, false, sizeof(visited));
		memset(map, 0, sizeof(map));

		int ans = 0;
		cin >> m >> n >> k;

		for (int i = 0; i < k; i++) {
			int y, x; cin >> x >> y;
			map[y][x] = 1;
			v.push_back({ y,x });
		}

		while (!v.empty()) {
			auto cur = v.back();
			v.pop_back();

			if (visited[cur.first][cur.second])
				continue;
			dfs(cur.first, cur.second);

			ans++;
		}
		cout << ans << '\n';
	}
}