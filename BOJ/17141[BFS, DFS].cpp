#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>
using namespace std;

const int INF = 0x3f3f3f;
int n, numOfVirus, ans = INF, map[50][50] = { 0, };
bool chk[50][50] = { false, };

const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

typedef pair<int, int> pii;
vector<pii> virus;
vector<pii> virus_pos;

int spread()
{
	int ret = 0, copy[50][50];
	bool visited[50][50] = { false, };

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			copy[i][j] = map[i][j];

	queue<pii> q;

	for (int i = 0; i < virus_pos.size(); i++) {
		q.push({ virus_pos[i] });
		visited[virus_pos[i].first][virus_pos[i].second] = true;
		copy[virus_pos[i].first][virus_pos[i].second] = 2;
	}

	while (!q.empty()) {
		int q_sz = q.size();
		
		while (q_sz--) {
			pii cur = q.front();
			q.pop();

			for (int dir = 0; dir < 4; dir++) {
				int ny = cur.first + dy[dir];
				int nx = cur.second + dx[dir];

				if (ny < 0 || nx < 0 || ny >= n || nx >= n)
					continue;
				if (!visited[ny][nx] && copy[ny][nx] == 0) {
					q.push({ ny,nx });
					visited[ny][nx] = true;
					copy[ny][nx] = 2;
				}
			}
		}
		ret++;
	}
	
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (copy[i][j] == 0)
				return INF;
	return ret - 1;
}

void put_virus(int cur, int cnt)
{
	if (cnt == numOfVirus) {
		ans = min(spread(), ans);
		return;
	}

	for (int i = cur; i < virus.size(); i++) {
		pii cur = virus[i];
		if (!chk[cur.first][cur.second]) {
			chk[cur.first][cur.second] = true;
			virus_pos.push_back({ cur.first, cur.second });
			
			put_virus(i, cnt + 1);
			
			chk[cur.first][cur.second] = false;
			virus_pos.pop_back();
		}
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	
	cin >> n >> numOfVirus;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			cin >> map[i][j];
			
			if (map[i][j] == 2) {
				virus.push_back({ i,j });
				map[i][j] = 0;
			}
		}

	put_virus(0, 0);
	cout << ((ans == INF) ? -1 : ans);
}