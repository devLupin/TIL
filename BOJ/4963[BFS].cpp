#include <iostream>
#include <queue>
#include <cstring>
#include <vector>
using namespace std;

int w, h;
int map[50][50] = { 0, };
bool visited[50][50];

const int dy[] = { -1,1,0,0,1,1,-1,-1 };
const int dx[] = { 0,0,-1,1,1,-1,1,-1 };

typedef pair<int, int> pii;
vector<pii> land;

int bfs()
{
	queue<pii> q;
	int ret = 0;

	for (auto it = land.begin(); it != land.end(); it++) {
		pii cur = (*it);

		if (visited[cur.first][cur.second])
			continue;

		q.push(cur);
		visited[cur.first][cur.second] = true;
		
		while (!q.empty()) {
			pii cur = q.front();
			q.pop();

			for (int dir = 0; dir < 8; dir++) {
				int ny = cur.first + dy[dir];
				int nx = cur.second + dx[dir];

				if (ny < 0 || nx < 0 || ny >= h || nx >= w)
					continue;
				if (visited[ny][nx] || map[ny][nx] == 0)
					continue;

				q.push({ ny, nx });
				visited[ny][nx] = true;
			}
		}

		ret++;
	}
	return ret;
}

int main(void) 
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	while (true) {
		memset(visited, false, sizeof(visited));
		memset(map, 0, sizeof(map));

		cin >> w >> h;

		if (w == 0 && h == 0)
			break;

		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++) {
				cin >> map[y][x];
				
				if (map[y][x] == 1)
					land.push_back({ y,x });
			}
		cout << bfs() << '\n';
		while (!land.empty()) land.pop_back();
	}
	return 0;
}