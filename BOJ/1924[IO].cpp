#include<iostream>
#include<string>
using namespace std;

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	
	string str[] = {"SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT" };
	int day[] = { 31,28,31,30,31,30,31,31,30,31,30,31 };
	
	int m, d, ans = 0; 
	cin >> m >> d;
	
	for (int i = 0; i < m - 1; i++)
		ans += day[i];
	ans += d;

	cout << str[ans % 7];
}