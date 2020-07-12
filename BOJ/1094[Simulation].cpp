#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int temp, ans = 0;
	cin >> temp;

	while (temp >= 1) {
		ans += (temp % 2);
		temp /= 2;
	}
	cout << ans;
}