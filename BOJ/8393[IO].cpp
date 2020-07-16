#include<iostream>
#include<string>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	
	int n, sum = 0;
	cin >> n;

	for (int i = 1; i <= n; i++)
		sum += i;
	cout << sum;
}