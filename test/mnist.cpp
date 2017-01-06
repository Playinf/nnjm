#include <fstream>
#include "mnist.h"

unsigned int msb2int(unsigned int data)
{
    unsigned int val;
    unsigned char* ptr;

    ptr = reinterpret_cast<unsigned char*>(&data);
    val = ptr[0] << 24;
    val |= ptr[1] << 16;
    val |= ptr[2] << 8;
    val |= ptr[3];

    return val;
}

mnist_label::mnist_label()
{
    label_vector = nullptr;
    label_number = 0;
}

mnist_label::~mnist_label()
{
    if (label_vector != nullptr)
        delete[] label_vector;
}

void mnist_label::read_file(const char* name)
{
    std::ifstream file;
    unsigned int integer;
    char* ptr = reinterpret_cast<char*>(&integer);

    file.open(name, std::ifstream::binary);

    if (file.fail())
        return;

    file.read(ptr, 4);

    /* magic number 0x0000801 */
    if (msb2int(integer) != 0x0000801)
        return;

    /* label number */
    file.read(ptr, 4);

    label_number = msb2int(integer);

    if (!label_number)
        return;

    label_vector = new unsigned char[label_number];

    for (unsigned int i = 0; i < label_number; i++) {
        file.read(reinterpret_cast<char*>(label_vector + i), 1);
    }
}

unsigned char* mnist_label::get_label() const
{
    return label_vector;
}

unsigned int mnist_label::get_size() const
{
    return label_number;
}

mnist_image::mnist_image()
{
    image_vector = nullptr;
    image_number = 0;
}

mnist_image::~mnist_image()
{
    if (image_vector != nullptr) {
        for (unsigned int i = 0; i < image_number; i++)
            delete[] image_vector[i];
        delete[] image_vector;
    }
}

void mnist_image::read_file(const char* name)
{
    std::ifstream file;
    unsigned int integer;
    char* ptr = reinterpret_cast<char*>(&integer);
    unsigned int size;

    file.open(name, std::ifstream::binary);

    if (file.fail())
        return;

    file.read(ptr, 4);

    /* magic number 0x0000803 */
    if (msb2int(integer) != 0x0000803)
        return;

    /* label number */
    file.read(ptr, 4);

    image_number = msb2int(integer);

    if (!image_number)
        return;

    file.read(ptr, 4);
    row_size = msb2int(integer);
    file.read(ptr, 4);
    column_size = msb2int(integer);

    size = row_size * column_size;
    image_vector = new pixel[image_number];

    for (unsigned int i = 0; i < image_number; i++) {
        image_vector[i] = new unsigned char[size];
        file.read(reinterpret_cast<char*>(image_vector[i]), size);
    }
}

mnist_image::pixel* mnist_image::get_image() const
{
    return image_vector;
}

unsigned int mnist_image::get_size() const
{
    return image_number;
}

unsigned int mnist_image::get_row_size() const
{
    return row_size;
}

unsigned int mnist_image::get_column_size() const
{
    return column_size;
}
