#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int arr[11] = { 0, };
	arr[1] = 1, arr[2] = 2, arr[3] = 4;

	for (int i = 4; i < 11; i++)
		arr[i] = arr[i - 3] + arr[i - 2] + arr[i - 1];

	int tc; cin >> tc;
	while (tc--) {
		int n; cin >> n;
		cout << arr[n] << '\n';
	}
}