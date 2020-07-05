#include <iostream>
#define MAX(A,B) ((A) > (B)) ? (A) : (B)
using namespace std;

int n, arr[300] = { 0, };
int DP[300] = { 0, };

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	cin >> n;
	for (int i = 0; i < n; i++)
		cin >> arr[i];

	DP[0] = arr[0];
	DP[1] = MAX(arr[0] + arr[1], arr[1]);
	DP[2] = MAX(arr[0] + arr[2], arr[1] + arr[2]);
	for (int i = 3; i < n; i++) {
		DP[i] = MAX(DP[i - 2] + arr[i], DP[i - 3] + arr[i - 1] + arr[i]);
	}

	cout << DP[n - 1];
}