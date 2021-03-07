#include <iostream>
using namespace std;

char MAP[500][500];
int DP[500][500];

int N, M;
const int dy[] = { -1,1,0,0 }, dx[] = { 0,0,-1,1 };

int get_dir(char c)
{
	if (c == 'U') return 0;
	else if (c == 'R') return 3;
	else if (c == 'D') return 1;
	else if (c == 'L') return 2;
}

int DFS(int r, int c)
{
	if (r < 0 || c < 0 || r >= N || c >= M) return 1;
	if (DP[r][c] != -1) return DP[r][c];

	DP[r][c] = 0;
	int Dir = get_dir(MAP[r][c]);

	DP[r][c] = DFS(r + dy[Dir], c + dx[Dir]);

	return DP[r][c];
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int ans = 0;

	cin >> N >> M;
	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < M; x++)
		{
			cin >> MAP[y][x];
			DP[y][x] = -1;
		}
	}

	for (int y = 0; y < N; y++)
		for (int x = 0; x < M; x++) {
			ans = ans + DFS(y, x);
		}
	cout << ans;
}