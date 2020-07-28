#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);

	char arr[10] = { 0, };
	cin >> arr;
	sort(arr, arr + strlen(arr), greater<int>());
	for (int i = 0; i < strlen(arr); i++)
		cout << arr[i];
	return 0;
}