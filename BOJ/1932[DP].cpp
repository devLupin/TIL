#include <iostream>
#define MAX(A,B) ((A) > (B)) ? (A) : (B)
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int arr[501][501] = { 0, };
	int n, max = -1;

	cin >> n;
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= i; j++)
			cin >> arr[i][j];

	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= i; j++) {
			if (j == 0)
				arr[i][j] += arr[i - 1][j];
			else if (i == j)
				arr[i][j] += arr[i - 1][j - 1];
			else
				arr[i][j] += (MAX(arr[i - 1][j - 1], arr[i - 1][j]));
			max = MAX(max, arr[i][j]);
		}
	cout << max;
}