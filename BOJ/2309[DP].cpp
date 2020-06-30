#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
using namespace std;

int arr[9];
int DP[9][9];
vector<int> ans;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int sum = 0;

	for (int i = 0; i < 9; i++) {
		cin >> arr[i];
		sum += arr[i];
	}

	memset(DP, 0, sizeof(DP));

	int erase1, erase2;
	bool chk = true;
	for (int i = 0; i < 9 && chk; i++)
		for (int j = 0; j < 9; j++) {
			if (!DP[i][j] && i != j)
				DP[i][j] = DP[j][i] = arr[i] + arr[j];

			if (sum - DP[i][j] == 100) {
				erase1 = i;
				erase2 = j;

				chk = false;
				break;
			}
		}

	for (int i = 0; i < 9; i++) {
		if (i == erase1 || i == erase2)
			continue;
		ans.push_back(arr[i]);
	}

	sort(ans.begin(), ans.end());
	for (int i = 0; i < ans.size(); i++) {
		cout << ans[i] << '\n';
	}
}