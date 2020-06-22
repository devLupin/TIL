#include <iostream>
#include <queue>
#include <cstring>
#include <vector>
#include <algorithm>
using namespace std;
typedef pair<int, int> pi;

int n;
char map[25][25];
bool visited[25][25];

vector<int> ans;

const int dy[] = { -1,1,0,0 };
const int dx[] = { 0,0,-1,1 };

void bfs(int sy, int sx) 
{
	int ret = 1;

	queue<pi> q;
	q.push({ sy, sx });
	visited[sy][sx] = true;

	while (!q.empty()) {
		pi cur = q.front();
		q.pop();

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.first + dy[dir];
			int nx = cur.second + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= n)
				continue;
			if (!visited[ny][nx] && map[ny][nx] == '1') {
				q.push({ ny,nx });
				visited[ny][nx] = true;
				ret++;
			}
		}
	}
	ans.push_back(ret);
}

int main(void)
{

	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n;

	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++)
			cin >> map[y][x];

	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++) {
			if (!visited[y][x] && map[y][x] == '1')
				bfs(y, x);
		}

	cout << ans.size() << '\n';
	sort(ans.begin(), ans.end());
	for (auto it = ans.begin(); it != ans.end(); it++)
		cout << (*it) << '\n';
}