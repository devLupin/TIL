#include <iostream>
#include <cstring>
using namespace std;

int n;
char map[100][100];
bool visited[100][100];

int normal, abnormal;

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

void dfs(int y, int x) {
	visited[y][x] = true;

	for (int dir = 0; dir < 4; dir++) {
		int ny = y + dy[dir];
		int nx = x + dx[dir];

		if (ny < 0 || nx < 0 || ny >= n || nx >= n)
			continue;
		if (!visited[ny][nx] && map[ny][nx] == map[y][x])
			dfs(ny, nx);
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++)
			cin >> map[y][x];

	normal = abnormal = 0;
	memset(visited, false, sizeof(visited));

	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++)
			if (!visited[y][x]) {
				normal++;
				dfs(y, x);
			}

	memset(visited, false, sizeof(visited));
	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++)
			if (map[y][x] == 'R' || map[y][x] == 'G')
				map[y][x] = 'R';

	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++)
			if (!visited[y][x]) {
				abnormal++;
				dfs(y, x);
			}

	cout << normal << ' ' << abnormal;
}