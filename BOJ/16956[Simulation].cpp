#include <iostream>
#include <queue>
#include <vector>
#include <cstring>
using namespace std;

typedef pair<int, int> pii;

int r, c, ans = 0x3f3f3f;
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };
char map[500][500], res[500][500];
bool visited[500][500], flag = false;
vector<pii> wolf;

bool check()
{
	for (int i = 0; i < wolf.size(); i++) {
		memset(visited, false, sizeof(visited));
		queue<pii> q;
		q.push({ wolf[i] });

		while (!q.empty()) {
			pii cur = q.front();
			q.pop();

			if (map[cur.first][cur.second] == 'S')
				return false;

			for (int dir = 0; dir < 4; dir++) {
				int ny = cur.first + dy[dir];
				int nx = cur.second + dx[dir];

				if (ny < 0 || nx < 0 || ny >= r || nx >= c)
					continue;
				if (!visited[ny][nx] && map[ny][nx] == '.' || map[ny][nx] == 'S') {
					q.push({ ny, nx });
					visited[ny][nx] = true;
				}
			}
		}
	}
	return true;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> r >> c;

	for(int i = 0; i < r; i++)
		for (int j = 0; j < c; j++) {
			cin >> map[i][j];
			
			if (map[i][j] == '.')
				map[i][j] = 'D';
			else if (map[i][j] == 'W')
				wolf.push_back({ i,j });
		}

	if (check()) {
		cout << '1' << '\n';
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++)
				cout << map[i][j];
			cout << '\n';
		}
	}
	else
		cout << '0';
}