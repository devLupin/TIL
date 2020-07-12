#include <iostream>
using namespace std;

int n, m;
char arr[50][50];

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int n, m, ans = 0x3f3f3f;
	cin >> n >> m;

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			cin >> arr[i][j];

	for (int i = 0; i <= n - 7; i++) {
		for (int j = 0; j <= m - 7; j++) {
			int cntB = 0, cntW = 0;

			for (int a = i; a < i + 8; a++) {
				for (int b = j; b < j + 8; b++) {
					if ((a + b) % 2 == 0)
						(arr[a][b] == 'B') ? cntW++ : cntB++;

					else
						(arr[a][b] == 'B') ? cntB++ : cntW++;
				}
			}
			ans = (ans > cntW) ? cntW : ans;
			ans = (ans > cntB) ? cntB : ans;
		}
	}
	cout << ans;
}