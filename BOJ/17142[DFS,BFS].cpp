#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>
using namespace std;

typedef pair<int, int> pii;
const int INF = 987654321;
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

int n, m, map[50][50], ans = INF;
bool visited[50][50] = { false, }, flag = false;

vector<pii> virus;

typedef struct {
	int y, x, cost;
}pos;

int spread()
{
	queue<pos> q;
	int copy[50][50] = { 0, };
	int ret = 0;

	for(int i=0; i<n; i++)
		for (int j = 0; j < n; j++) {
			copy[i][j] = map[i][j];

			if (copy[i][j] == 3)
				q.push({ i,j,0 });
		}

	while (!q.empty())
	{
		int q_sz = q.size();
		while (q_sz--) {
			auto cur = q.front();
			q.pop();

			for (int dir = 0; dir < 4; dir++) {
				int ny = cur.y + dy[dir];
				int nx = cur.y + dx[dir];

				if (ny < 0 || nx < 0 || ny >= n || nx >= n)
					continue;

				if (copy[ny][nx] == 0) {
					ret = max(cur.cost + 1, ret);
					q.push({ ny, nx, cur.cost + 1 });
					copy[ny][nx] = 3;
				}
				else if (copy[ny][nx] == 2) {
					ret = max(cur.cost, ret);
					q.push({ ny, nx, cur.cost});
					copy[ny][nx] = 3;
				}
				cout << ret << '\n';
				visited[ny][nx] = true;
			}
		}
	}

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			if (copy[i][j] == 0)
				return -1;
	
	return ret;
}

void put_virus(int start, int cnt)
{
	if (cnt == m) {
		int comp = spread();
		ans = (comp > 0) ? min(ans, comp) : ans;
		return;
	}

	for (int i = start; i < virus.size(); i++) {
		auto cur = virus[i];

		if (map[cur.first][cur.second] == 2) {
			map[cur.first][cur.second] = 3;
			put_virus(i, cnt + 1);
			map[cur.first][cur.second] = 2;
		}
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n >> m;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			cin >> map[i][j];
			
			if (map[i][j] == 2)
				virus.push_back({ i,j });
		}

	put_virus(0, 0);
	cout << ans;
}