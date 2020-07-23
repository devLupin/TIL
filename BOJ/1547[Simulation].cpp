#include <iostream>
#include <string>
using namespace std;

int arr[4] = { 0, 1, 2, 3 };
void swap(int x, int y)
{
	int x_idx, y_idx;
	for (int i = 1; i < 4; i++) {
		if (arr[i] == x)
			x_idx = i;
		else if (arr[i] == y)
			y_idx = i;
	}
	int temp = arr[x_idx];
	arr[x_idx] = arr[y_idx];
	arr[y_idx] = temp;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	int m; cin >> m;
	for (int i = 0; i < m; i++) {
		int x, y; cin >> x >> y;
		swap(x, y);
	}
	cout << arr[1];
}