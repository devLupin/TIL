#include <iostream>
#include <deque>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	int a, b, c, ans = 0, arr[101] = { 0, };
	cin >> a >> b >> c;

	for (int i = 0; i < 3; i++) {
		int x, y; cin >> x >> y;
		for (int j = x; j < y; j++)
			arr[j]++;
	}

	for (int i = 1; i < 101; i++) {
		switch (arr[i]) {
		case 1:
			ans += a;
			break;
		case 2:
			ans += (arr[i] * b);
			break;
		case 3:
			ans += (arr[i] * c);
			break;
		}
	}
	cout << ans;
}