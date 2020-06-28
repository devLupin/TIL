#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
using namespace std;

typedef struct {
	int y, x;
}pos;

int n, m, k;
int map[100][100];
bool visited[100][100];
int cnt;
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

void dfs(int y, int x)
{
	visited[y][x] = true;
	cnt++;

	for (int dir = 0; dir < 4; dir++) {
		int ny = y + dy[dir];
		int nx = x + dx[dir];

		if (ny < 0 || nx < 0 || ny >= n || nx >= m)
			continue;
		if (map[ny][nx] == 0 && !visited[ny][nx])
			dfs(ny, nx);
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	memset(visited, false, sizeof(visited));
	memset(map, 0, sizeof(map));

	cin >> n >> m >> k;
	for (int i = 0; i < k; i++) {
			int x1, y1, x2, y2;
			cin >> x1 >> y1 >> x2 >> y2;
			
			for (int y = y1; y < y2; y++)
				for (int x = x1; x < x2; x++)
					map[y][x] = 1;
		}

	vector<int> ans;

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++) {
			if (!visited[i][j] && map[i][j] == 0) {
				cnt = 0;
				dfs(i, j);
				ans.push_back(cnt);
			}
		}

	sort(ans.begin(), ans.end());
	cout << ans.size() << '\n';
	for (auto it = ans.begin(); it != ans.end(); it++)
		cout << (*it) << " ";
}