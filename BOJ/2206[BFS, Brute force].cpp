#include <iostream>
#include <queue>
#include <cstring>
#define MIN(A,B) (A) < (B) ? (A) : (B)
using namespace std;

typedef struct {
	int y, x, dist, wall;
}pos;

char map[1000][1000];
bool visited[1000][1000][2];

int n, m;
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };



/*
 이 문제의 경우 DFS로 벽을 하나씩 부수면서 탐색을 진행하면 시간초과가 난다.
 예제는 나오지만 많은 벽이 존재하는 경우 엄청난 시간을 잡아먹어 DFS,BFS를 합치면 N^4 임.
 따라서 3차원배열의 마지막 사이즈를 2로 잡아 벽을 부쉈는지 또는 아닌지 확인하는 작업을 거쳤다.
*/



int bfs()
{
	memset(visited, false, sizeof(visited));

	queue<pos> q;
	q.push({ 0,0,1,1});
	visited[0][0][0] = true;

	while (!q.empty()) {
		pos cur = q.front();
		q.pop();

		if (cur.y == n - 1 && cur.x == m - 1)
			return cur.dist;

		for (int dir = 0; dir < 4; dir++) {
			int ny = cur.y + dy[dir];
			int nx = cur.x + dx[dir];

			if (ny < 0 || nx < 0 || ny >= n || nx >= m)
				continue;
			if (visited[ny][nx][cur.wall])
				continue;
			if (map[ny][nx] == '0') {
				q.push({ ny,nx,cur.dist + 1,cur.wall });
				visited[ny][nx][cur.wall] = true;
			}
			if (map[ny][nx] == '1' && cur.wall > 0) {
				q.push({ ny,nx,cur.dist + 1, 0 });
				visited[ny][nx][cur.wall] = true;
			}
		}
	}
	return -1;
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