#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

typedef struct {
	int y, x;
}pos;

int n, h = -1;
int map[101][101];
bool visited[101][101];

const int dy[] = { -1,1,0,0 };
const int dx[] = { 0,0,-1,1 };

int bfs(int max_height)
{
	memset(visited, false, sizeof(visited));
	
	int ret = 0;
	queue<pos> q;
	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++) {
			if (!visited[y][x] && map[y][x] >= max_height) {
				
				ret++;

				q.push({ y,x });
				visited[y][x] = true;
				while (!q.empty()) {
					pos cur = q.front();
					q.pop();

					for (int dir = 0; dir < 4; dir++) {
						int ny = cur.y + dy[dir];
						int nx = cur.x + dx[dir];

						if (ny < 0 || nx < 0 || ny >= n || nx >= n)
							continue;
						if (!visited[ny][nx] && map[ny][nx] >= max_height) {
							q.push({ ny,nx });
							visited[ny][nx] = true;
						}
					}
				}
			}
		}
			
	return ret;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int ans = 0;
	cin >> n;

	for (int y = 0; y < n; y++)
		for (int x = 0; x < n; x++) {
			cin >> map[y][x];
			h = (h < map[y][x]) ? map[y][x] : h;
		}
	
	for (int i = 1; i <= h; i++) {
		int comp = bfs(i);
		ans = (ans < comp) ? comp : ans;
	}
	cout << ans;
}