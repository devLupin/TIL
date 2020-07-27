#include <iostream>
#include <set>
using namespace std;

set<int> st;
int map[5][5] = { 0, };
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

void dfs(int y, int x, int cnt, int sum)
{
	if (cnt == 5) {
		st.insert(sum);
		return;
	}

	for (int dir = 0; dir < 4; dir++) {
		int ny = y + dy[dir];
		int nx = x + dx[dir];

		if (ny < 0 || nx < 0 || ny >= 5 || nx >= 5)
			continue;
		dfs(ny, nx, cnt + 1, sum * 10 + map[ny][nx]);
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			cin >> map[i][j];

	

	dfs(0, 0, 0, 0);

	cout << st.size();
}