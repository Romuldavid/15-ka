#include <SFML/Graphics.hpp>
#include <time.h>
using namespace sf;

int main()
{
    srand(time(0));

    RenderWindow window(VideoMode(512, 512), "15-ka");
    window.setFramerateLimit(60);

    Texture t;
    int a = rand()%5 + 1;
    switch (a)
    {
    case 1:
        t.loadFromFile("Paint/Pro/1.png");
        break;
    case 2:
        t.loadFromFile("Paint/Pro/2.png");
        break;
    case 3:
        t.loadFromFile("Paint/Pro/3.png");
        break;
    case 4:
        t.loadFromFile("Paint/Pro/4.png");
        break;
    case 5:
        t.loadFromFile("Paint/Pro/5.png");
        break;
    
    }
    // t.loadFromFile("Paint/15.png");
    Sprite s[17];
    int w = 128;
    int grid[6][6] = { 0 };
    int n = 0;
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
        {
            n++;
            s[n].setTexture(t);
            s[n].setTextureRect(IntRect(i * w, j * w, w, w));
            grid[i + 1][j + 1] = n;
        }

    while(window.isOpen())
    {
        Event event;
        while(window.pollEvent(event))
        {
            if(event.type == Event::Closed)
            window.close();

            if(event.type == Event::MouseButtonPressed)
                if(event.key.code == Mouse::Left)
                {
                    Vector2i pos = Mouse::getPosition(window);

                    int x = pos.x / w + 1;
                    int y = pos.y / w + 1;

                    int dx = 0;
                    int dy = 0;

                    int b;
                    if(a != 2 && a !=5)
                        b = 16;
                    else if( a == 2 || a == 5)
                        b = 4;

                    // for X
                    if(grid[x + 1][y] == b)
                    {
                        dx = 1;
                        dy = 0;
                    }
                    if(grid[x - 1][y] == b)
                    {
                        dx = -1;
                        dy = 0;
                    }

                    //for Y
                    if(grid[x][y - 1] == b)
                    {
                        dx = 0;
                        dy = -1;
                    }
                    if(grid[x][y + 1] == b)
                    {
                        dx = 0;
                        dy = 1;
                    }

                    n = grid[x][y];
                    grid[x][y] = b;
                    grid[x + dx][y + dy] = n;

                    s[b].move(-dx * w, -dy * w);
                    float speed = 20;
                    for(int i = 0; i < w; i += speed)
                    {
                        s[n].move(speed * dx, speed * dy);
                        window.draw(s[b]);
                        window.draw(s[n]);
                        window.display();
                    }
                }
        }

        window.clear(Color::White);
        for(int i = 0; i < 4; i++)
            for(int j = 0; j < 4; j++)
            {
                n = grid[i + 1][j + 1];
                s[n].setPosition(i * w, j * w);
                window.draw(s[n]);
            }
        window.display();
    }
}