#pragma OPENCL EXTENSION cl_amd_printf : enable

float area(float2 c1, float2 c2, float2 c3)
{

    return 0.5f * fabs(
                      (c1.x * (c2.y - c3.y)) +
                      (c2.x * (c3.y - c1.y)) +
                      (c3.x * (c1.y - c2.y)));
}

bool inside(float2 uvhit, float2 uv1, float2 uv2, float2 uv3)
{
    return fabs(area(uv1, uv2, uv3) -
                (area(uvhit, uv1, uv2) + area(uvhit, uv1, uv3) + area(uvhit, uv2, uv3))) < 0.001;
}

#define ADDR ((y_addr * xsize) + x_addr)
#define ADDR3(x, y, z) (x + (y * lxsize) + (z * lxsize * lysize))
#define ADDRBUFFER2 ((ly * lxsize) + lx)

// Simple ray tracing to detect an object
// no reflections
// camera looks at -z and is at 0,0
__kernel void trace(__global float4 *in_data, __global float4 *color_data, __global float4 *out_data, __local float4 *subpixel_array,
                    __local int *hitbufferi, __local float *hitbuffert,
                    __local float4 *R0buffer, __local float4 *Rdbuffer, __local float4 *Ribuffer,
                    __local float4 *Nbuffer,
                    int in_data_len, int xsize, int ysize, float focal_len)
{

    uint x_addr;
    uint y_addr;

    uint lx;
    uint ly;
    uint lz;

    uint lindex;

    uint lxsize;
    uint lysize;
    uint lzsize;

    float xcast;
    float ycast;
    float len;

    // x and y positions of the pixel
    x_addr = get_group_id(0);
    y_addr = get_group_id(1);

    // position of the subpixel
    lx = get_local_id(0);
    ly = get_local_id(1);

    // hit buffer index
    lz = get_local_id(2);

    // num of subpixels
    lxsize = get_local_size(0);
    lysize = get_local_size(1);
    // num of hitbuffer
    lzsize = get_local_size(2);

    hitbufferi[ADDR3(lx, ly, lz)] = -1;
    hitbuffert[ADDR3(lx, ly, lz)] = -1.0;

    xcast = ((int)x_addr) - (xsize / 2) + (((float)lx) / ((float)lxsize));
    ycast = ((int)y_addr) - (ysize / 2) + (((float)ly) / ((float)lysize));

    if (lz == 0)
    {
        float4 Rd;
        Rd.x = xcast;
        Rd.y = ycast;
        Rd.z = -focal_len;
        Rd.w = 0.0;

        Rd = normalize(Rd);
        Rdbuffer[ADDRBUFFER2] = Rd;

        float4 R0 = (float4)(0.0);
        R0buffer[ADDRBUFFER2] = R0;
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    /*
    As our ray passes through the origin,
    our ray function is

    x = xcast * t
    y = ycast * t
    z = flen * t

    */

    // TODO iterate max_bounces times to get reflections etc {

    /*
     iterate over all polygons
     found hits are stored over hitbuffer
    using a hitbuffer should be more efficient with large number of polygons
    */

    for (int i = lz; i < in_data_len; i++)
    {
        int _i = i * 3;
        float3 c1 = in_data[_i].xyz;
        float3 c2 = in_data[_i + 1].xyz;
        float3 c3 = in_data[_i + 2].xyz;

        /*
        Define the plane as A B C D
        */
        float3 v1 = c2 - c1;
        float3 v2 = c3 - c1;

        float4 normal = (float4)(fast_normalize(cross(v1, v2)), 0.0f);
        /*
        Now our plane equation is
        normal.x * (X - c1.x) + normal.y * (Y - c1.y) + normal.z * (Z - c1.z) = 0
        normal.x * X + normal.y * Y + normal.z * Z - (normal.x * c1.x + normal.y + c1.y + normal.z * c1.z)
         */

        float W = ((normal.x * c1.x) + (normal.y * c1.y) + (normal.z * c1.z));

        float vd = dot(normal, Rdbuffer[ADDRBUFFER2]);
        if (vd < 0)
        {
        }
        else
        {

            float v0 = W;

            float t = v0 / vd;

            if (t < 0)
            {
            }
            else
            {

                // we are hitting the plane
                // check if it is the closest hit

                if (t > hitbuffert[ADDR3(lx, ly, lz)] && hitbuffert[ADDR3(lx, ly, lz)] > 0)
                {
                    // occuluded edge
                }
                else
                {

                    // point of intersection
                    float3 ri = (float3)(Rdbuffer[ADDRBUFFER2].x * t, Rdbuffer[ADDRBUFFER2].y * t, Rdbuffer[ADDRBUFFER2].z * t);

                    // check if the point on plane is inside the triangle

                    // get rid of the biggest magnitude plane axis to get uv coords
                    float2 uv_hit;
                    float2 uv1;
                    float2 uv2;
                    float2 uv3;

                    float mmax = maxmag(normal.x, maxmag(normal.y, normal.z));
                    if (mmax == normal.x)
                    {
                        uv_hit = (float2)(ri.y, ri.z);
                        uv1 = (float2)(c1.y, c1.z);
                        uv2 = (float2)(c2.y, c2.z);
                        uv3 = (float2)(c3.y, c3.z);
                    }
                    if (mmax == normal.y)
                    {
                        uv_hit = (float2)(ri.x, ri.z);
                        uv1 = (float2)(c1.x, c1.z);
                        uv2 = (float2)(c2.x, c2.z);
                        uv3 = (float2)(c3.x, c3.z);
                    }
                    if (mmax == normal.z)
                    {
                        uv_hit = (float2)(ri.x, ri.y);
                        uv1 = (float2)(c1.x, c1.y);
                        uv2 = (float2)(c2.x, c2.y);
                        uv3 = (float2)(c3.x, c3.y);
                    }

                    // check if uv_hit is inside the triangle formed by other uvs
                    if (inside(uv_hit, uv1, uv2, uv3))
                    {
                        // we are colliding with this polygon.
                        // replace our buffer i, buffer t and normal values
                        hitbufferi[ADDR3(lx, ly, lz)] = i;
                        hitbuffert[ADDR3(lx, ly, lz)] = t;
                        Nbuffer[ADDR3(lx, ly, lz)] = normal;
                        Ribuffer[ADDR3(lx, ly, lz)] = (float4)(ri, 0.0f);
                    }
                    else
                    {
                    }
                }
            }
        }
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // combine all elems in zbuffer
    if (lz == 0)
    {
        float mint = hitbuffert[ADDR3(lx, ly, 0)];
        int minz = 0;
        for (int i = 1; i < lzsize; i++)
        {
            float ht = hitbuffert[ADDR3(lx, ly, i)];
            if (mint < 0 || (mint > ht && ht > 0))
            {
                mint = ht;
                minz = i;
            }
        }

        // the hit polygon is index minz, we can access the normal and everything with i
        // TODO here, compute reflection, update Rd, R0, and other things related to reflections etc

        hitbuffert[ADDR3(lx, ly, 0)] = hitbuffert[ADDR3(lx, ly, minz)];
        hitbufferi[ADDR3(lx, ly, 0)] = hitbufferi[ADDR3(lx, ly, minz)];
        // I don't know how to set reflected ray, ie Rd
        R0buffer[ADDRBUFFER2] = Ribuffer[ADDR3(lx, ly, minz)];
    }

    // }, assume max bounces end here
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (lz == 0)
    {
        // compute subpixel color
        // for not set to 1 if there was any hit at all
        if (hitbufferi[ADDR3(lx, ly, 0)] != -1)
        {
            subpixel_array[ADDRBUFFER2] = color_data[hitbufferi[ADDR3(lx, ly, 0)]];
        }
        else
        {
            subpixel_array[ADDRBUFFER2] = (float4)(0.0);
        }
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    // approximate pixel color from subpixels
    // TODO we can parallelize this to sum on x first then sum on y later
    if (lx == 0 && ly == 0 && lz == 0)
    {
        float4 color = (float4)(0.0f);
        for (int i = 0; i < lxsize; i++)
        {
            for (int j = 0; j < lysize; j++)
            {
                color += subpixel_array[(j * lxsize) + i];
            }
        }
        color /= (lxsize * lysize);
        out_data[ADDR] = color;
    }
}
