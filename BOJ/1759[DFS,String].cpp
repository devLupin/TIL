#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

char arr[16];
bool visited[16] = { false, };
int l, c;
string ans = "";

void dfs(char cur, int cnt)
{
	if (cnt == l) {
		int mo = 0, ja = 0;
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == 'a' || ans[i] == 'e' || ans[i] == 'i' || ans[i] == 'o' || ans[i] == 'u' ||
				ans[i] == 'A' || ans[i] == 'E' || ans[i] == 'I' || ans[i] == 'O' || ans[i] == 'U') {
				mo++;
			}
			else
				ja++;
		}
		if(mo >= 1 && ja >= 2)
			cout << ans << '\n';
		return;
	}
	string temp = ans;

	for (int i = 0; i < c; i++) {
		if (!visited[i] && cur < arr[i]) {
			visited[i] = true;
			ans += arr[i];
			dfs(arr[i], cnt + 1);
			visited[i] = false;
			ans = temp;
		}
	}
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> l >> c;
	for (int i = 0; i < c; i++)
		cin >> arr[i];

	sort(arr, arr + c);

	for (int i = 0; i < c; i++) {
		visited[i] = true;
		ans += arr[i];
		dfs(arr[i], 1);
		ans = "";
		visited[i] = false;
	}
}