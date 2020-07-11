#include <iostream>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int arr[5] = { 0, };
	
	int big = 0;
	for (int i = 1; i < 4; i++) {
		int a, b; cin >> a >> b;
		arr[i - 1] -= a;
		arr[i] += arr[i-1] + b;
		big = (arr[i] > big) ? arr[i] : big;
	}
	cout << big;
}