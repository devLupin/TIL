#include <iostream>
#include <queue>
#include <cstring>
using namespace std;

typedef struct {
	int y, x, w;
}pos;

int l;
pos src, dest;

const int dy[] = { 1,2,2,1,-1,-2,-2,-1 };
const int dx[] = { -2,-1,1,2,2,1,-1,-2 };

int map[301][301];
bool visited[301][301];

int bfs()
{
	memset(visited, false, sizeof(visited));
	src.w = 0;
	visited[src.y][src.x] = true;

	queue<pos> q;
	q.push(src);
	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		if (cur.y == dest.y && cur.x == dest.x)
			return cur.w;

		for (int dir = 0; dir < 8; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= l || nx >= l)
				continue;
			if (!visited[ny][nx]) {
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

	int tc; cin >> tc;
	while (tc--) {
		cin >> l;
		cin >> src.y >> src.x >> dest.y >> dest.x;

		cout << bfs() << '\n';
	}
}