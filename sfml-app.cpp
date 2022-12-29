#include <SFML/Graphics.hpp>
using namespace sf;

int main()
{
    RenderWindow window(VideoMode(512, 512), "15-ka");

    Texture t;
    t.loadFromFile("Paint/15.png");
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

                    // for X
                    if(grid[x + 1][y] == 16)
                    {
                        dx = 1;
                        dy = 0;
                    }
                    if(grid[x - 1][y] == 16)
                    {
                        dx = -1;
                        dy = 0;
                    }

                    //for Y
                    if(grid[x][y - 1] == 16)
                    {
                        dx = 0;
                        dy = -1;
                    }
                    if(grid[x][y + 1] == 16)
                    {
                        dx = 0;
                        dy = 1;
                    }

                    n = grid[x][y];
                    grid[x][y] = 16;
                    grid[x + dx][y + dy] = n;
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